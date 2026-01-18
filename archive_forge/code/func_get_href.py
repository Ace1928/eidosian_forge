import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import XmlResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeState
def get_href(element, rel):
    """
    Search a RESTLink element in the :class:`AbiquoResponse`.

    Abiquo, as a REST API, it offers self-discovering functionality.
    That means that you could walk through the whole API only
    navigating from the links offered by the entities.

    This is a basic method to find the 'relations' of an entity searching into
    its links.

    For instance, a Rack entity serialized as XML as the following::

        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <rack>
         <link href="http://host/api/admin/datacenters/1"
            type="application/vnd.abiquo.datacenter+xml" rel="datacenter"/>
         <link href="http://host/api/admin/datacenters/1/racks/1"
            type="application/vnd.abiquo.rack+xml" rel="edit"/>
         <link href="http://host/api/admin/datacenters/1/racks/1/machines"
            type="application/vnd.abiquo.machines+xml" rel="machines"/>
         <haEnabled>false</haEnabled>
         <id>1</id>
         <longDescription></longDescription>
         <name>racacaca</name>
         <nrsq>10</nrsq>
         <shortDescription></shortDescription>
         <vlanIdMax>4094</vlanIdMax>
         <vlanIdMin>2</vlanIdMin>
         <vlanPerVdcReserved>1</vlanPerVdcReserved>
         <vlansIdAvoided></vlansIdAvoided>
        </rack>

    offers link to datacenters (rel='datacenter'), to itself (rel='edit') and
    to the machines defined in it (rel='machines')

    A call to this method with the 'rack' element using 'datacenter' as 'rel'
    will return:

    'http://10.60.12.7:80/api/admin/datacenters/1'

    :type  element: :class:`xml.etree.ElementTree`
    :param element: Xml Entity returned by Abiquo API (required)
    :type      rel: ``str``
    :param     rel: relation link name
    :rtype:         ``str``
    :return:        the 'href' value according to the 'rel' input parameter
    """
    links = element.findall('link')
    for link in links:
        if link.attrib['rel'] == rel:
            href = link.attrib['href']
            needle = '/api/'
            url_path = urlparse.urlparse(href).path
            index = url_path.find(needle)
            result = url_path[index + len(needle) - 1:]
            return result