from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def handle_define(module, v):
    """ handle `command: define` """
    xml = module.params.get('xml', None)
    guest = module.params.get('name', None)
    autostart = module.params.get('autostart', None)
    mutate_flags = module.params.get('mutate_flags', [])
    if not xml:
        module.fail_json(msg="define requires 'xml' argument")
    try:
        incoming_xml = etree.fromstring(xml)
    except etree.XMLSyntaxError:
        module.fail_json(msg='given XML is invalid')
    domain_name = incoming_xml.findtext('./name')
    if domain_name is not None:
        if guest is not None and domain_name != guest:
            module.fail_json("given 'name' parameter does not match name in XML")
    else:
        if guest is None:
            module.fail_json("missing 'name' parameter and no name provided in XML")
        domain_name = guest
        etree.SubElement(incoming_xml, 'name').text = domain_name
    if domain_name == '':
        module.fail_json(msg='domain name cannot be an empty string')
    res = dict()
    try:
        existing_domain = v.get_vm(domain_name)
        existing_xml_raw = existing_domain.XMLDesc(libvirt.VIR_DOMAIN_XML_INACTIVE)
        existing_xml = etree.fromstring(existing_xml_raw)
    except VMNotFound:
        existing_domain = None
        existing_xml_raw = None
        existing_xml = None
    if existing_domain is not None:
        incoming_uuid = incoming_xml.findtext('./uuid')
        existing_uuid = existing_domain.UUIDString()
        if incoming_uuid is not None and incoming_uuid != existing_uuid:
            module.fail_json(msg='attempting to re-define domain %s/%s with a different UUID: %s' % (domain_name, existing_uuid, incoming_uuid))
        else:
            if 'ADD_UUID' in mutate_flags and incoming_uuid is None:
                etree.SubElement(incoming_xml, 'uuid').text = existing_uuid
            existing_devices = existing_xml.find('./devices')
            if 'ADD_MAC_ADDRESSES' in mutate_flags:
                for interface in incoming_xml.xpath('./devices/interface[not(mac) and alias]'):
                    search_alias = interface.find('alias').get('name')
                    xpath = "./interface[alias[@name='%s']]" % search_alias
                    try:
                        matched_interface = existing_devices.xpath(xpath)[0]
                        existing_devices.remove(matched_interface)
                        etree.SubElement(interface, 'mac', {'address': matched_interface.find('mac').get('address')})
                    except IndexError:
                        module.warn('Could not match interface %i of incoming XML by alias %s.' % (interface.getparent().index(interface) + 1, search_alias))
            if 'ADD_MAC_ADDRESSES_FUZZY' in mutate_flags:
                similar_interface_counts = {}

                def get_interface_count(_type, source=None):
                    key = (_type, source if _type != 'user' else None)
                    if key not in similar_interface_counts:
                        similar_interface_counts[key] = 1
                    else:
                        similar_interface_counts[key] += 1
                    return similar_interface_counts[key]
                for interface in incoming_xml.xpath('./devices/interface'):
                    _type = interface.get('type')
                    if interface.find('mac') is not None and interface.find('alias') is not None:
                        continue
                    if _type not in INTERFACE_SOURCE_ATTRS:
                        module.warn("Skipping fuzzy MAC matching for interface %i of incoming XML: unsupported interface type '%s'." % (interface.getparent().index(interface) + 1, _type))
                        continue
                    source_attr = INTERFACE_SOURCE_ATTRS[_type]
                    source = interface.find('source').get(source_attr) if source_attr else None
                    similar_count = get_interface_count(_type, source)
                    if interface.find('mac') is not None:
                        continue
                    if source:
                        xpath = "./interface[@type='%s' and source[@%s='%s']]" % (_type, source_attr, source)
                    else:
                        xpath = "./interface[@type = '%s']" % source_attr
                    matching_interfaces = existing_devices.xpath(xpath)
                    try:
                        matched_interface = matching_interfaces[similar_count - 1]
                        etree.SubElement(interface, 'mac', {'address': matched_interface.find('./mac').get('address')})
                    except IndexError:
                        module.warn('Could not fuzzy match interface %i of incoming XML.' % (interface.getparent().index(interface) + 1))
    try:
        domain_xml = etree.tostring(incoming_xml).decode()
        domain = v.define(domain_xml)
        if existing_domain is not None:
            new_xml = domain.XMLDesc(libvirt.VIR_DOMAIN_XML_INACTIVE)
            if existing_xml_raw != new_xml:
                res.update({'changed': True, 'change_reason': 'domain definition changed', 'diff': {'before': existing_xml_raw, 'after': new_xml}})
        else:
            res.update({'changed': True, 'created': domain.name()})
    except libvirtError as e:
        module.fail_json(msg='libvirtError: %s' % e.get_error_message())
    except Exception as e:
        module.fail_json(msg='an unknown error occured: %s' % e)
    if autostart is not None and v.autostart(domain_name, autostart):
        res.update({'changed': True, 'change_reason': 'autostart'})
    return res