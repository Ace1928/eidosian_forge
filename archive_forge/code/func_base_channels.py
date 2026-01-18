from __future__ import absolute_import, division, print_function
import ssl
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xmlrpc_client
def base_channels(client, session, sys_id):
    basechan = client.channel.software.listSystemChannels(session, sys_id)
    try:
        chans = [item['label'] for item in basechan]
    except KeyError:
        chans = [item['channel_label'] for item in basechan]
    return chans