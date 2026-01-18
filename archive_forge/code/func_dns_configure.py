from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def dns_configure(self):
    dns_add_remove = False
    modify = {}
    existing_dns = {}
    existing_dns_server = []
    existing_dns_ip = []
    if self.module.check_mode:
        self.changed = True
        return
    dns_data = self.get_existing_dnsservers()
    self.log('dns_data=%s', dns_data)
    if self.dnsip and self.dnsname or (self.dnsip == '' and self.dnsname == ''):
        for server in dns_data:
            existing_dns_server.append(server['name'])
            existing_dns_ip.append(server['IP_address'])
            existing_dns[server['name']] = server['IP_address']
        for name, ip in zip(self.dnsname, self.dnsip):
            if name == 'None':
                self.log(' Empty DNS configuration is provided.')
                return
            if name in existing_dns:
                if existing_dns[name] != ip:
                    self.log('update, diff IP.')
                    modify[name] = ip
                else:
                    self.log('no update, same IP.')
        if set(existing_dns_server).symmetric_difference(set(self.dnsname)):
            dns_add_remove = True
    if modify:
        for item in modify:
            self.restapi.svc_run_command('chdnsserver', {'ip': modify[item]}, [item])
        self.changed = True
        self.message += ' DNS %s modified.' % modify
    if dns_add_remove:
        to_be_added, to_be_removed = (False, False)
        to_be_removed = list(set(existing_dns_server) - set(self.dnsname))
        if to_be_removed:
            for item in to_be_removed:
                self.restapi.svc_run_command('rmdnsserver', None, [item])
                self.changed = True
            self.message += ' DNS server %s removed.' % to_be_removed
        to_be_added = list(set(self.dnsname) - set(existing_dns_server))
        to_be_added_ip = list(set(self.dnsip) - set(existing_dns_ip))
        if any(to_be_added):
            for dns_name, dns_ip in zip(to_be_added, to_be_added_ip):
                if dns_name:
                    self.log('%s %s', dns_name, dns_ip)
                    self.restapi.svc_run_command('mkdnsserver', {'name': dns_name, 'ip': dns_ip}, cmdargs=None)
                    self.changed = True
            self.message += ' DNS server %s added.' % to_be_added
    elif not modify:
        self.log('No DNS Changes')