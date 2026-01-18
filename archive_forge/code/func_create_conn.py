import collections
from unittest import mock
import uuid
from openstack.network.v2 import vpn_endpoint_group as vpn_epg
from openstack.network.v2 import vpn_ike_policy as vpn_ikep
from openstack.network.v2 import vpn_ipsec_policy as vpn_ipsecp
from openstack.network.v2 import vpn_ipsec_site_connection as vpn_sitec
from openstack.network.v2 import vpn_service
@staticmethod
def create_conn(attrs=None):
    """Create a fake IPsec conn.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A Dictionary with id, name, peer_address, auth_mode, status,
            project_id, peer_cidrs, vpnservice_id, ipsecpolicy_id,
            ikepolicy_id, mtu, initiator, admin_state_up, description,
            psk, route_mode, local_id, peer_id, local_ep_group_id,
            peer_ep_group_id
        """
    attrs = attrs or {}
    conn_attrs = {'id': 'ipsec-site-conn-id-' + uuid.uuid4().hex, 'name': 'my-ipsec-site-conn-' + uuid.uuid4().hex, 'peer_address': '192.168.2.10', 'auth_mode': '', 'status': '', 'project_id': 'project-id-' + uuid.uuid4().hex, 'peer_cidrs': [], 'vpnservice_id': 'vpnservice-id-' + uuid.uuid4().hex, 'ipsecpolicy_id': 'ipsecpolicy-id-' + uuid.uuid4().hex, 'ikepolicy_id': 'ikepolicy-id-' + uuid.uuid4().hex, 'mtu': 1500, 'initiator': 'bi-directional', 'admin_state_up': True, 'description': 'my-vpn-connection', 'psk': 'abcd', 'route_mode': '', 'local_id': '', 'peer_id': '192.168.2.10', 'local_ep_group_id': 'local-ep-group-id-' + uuid.uuid4().hex, 'peer_ep_group_id': 'peer-ep-group-id-' + uuid.uuid4().hex}
    conn_attrs.update(attrs)
    return vpn_sitec.VpnIPSecSiteConnection(**conn_attrs)