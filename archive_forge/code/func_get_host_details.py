from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_host_details(self, host_id=None, host_name=None):
    """ Get details of a given host """
    host_id_or_name = host_id if host_id else host_name
    try:
        LOG.info('Getting host %s details', host_id_or_name)
        if host_id:
            host_details = self.unity.get_host(_id=host_id)
            if host_details.name is None:
                return None
        if host_name:
            ' get the count of hosts with same host_name '
            host_count = self.get_host_count(host_name)
            if host_count < 1:
                return None
            elif host_count > 1:
                error_message = 'Duplicate hosts found: There are ' + host_count + ' hosts(s) with the same host_name: ' + host_name
                LOG.error(error_message)
                self.module.fail_json(msg=error_message)
            else:
                host_details = self.unity.get_host(name=host_name)
        return host_details
    except utils.HttpError as e:
        if e.http_status == 401:
            msg = 'Incorrect username or password provided.'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        else:
            msg = 'Got HTTP Connection Error while getting host details %s : Error %s ' % (host_id_or_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    except utils.UnityResourceNotFoundError as e:
        error_message = 'Failed to get details of host {0} with error {1}'.format(host_id_or_name, str(e))
        LOG.error(error_message)
        return None
    except Exception as e:
        error_message = 'Got error %s while getting details of host %s' % (str(e), host_id_or_name)
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)