import logging
import socket
import struct
import traceback
from socket import IPPROTO_TCP, TCP_NODELAY
from eventlet import semaphore
from os_ken.lib.packet import bgp
from os_ken.lib.packet.bgp import AS_TRANS
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.packet.bgp import BGPOpen
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPKeepAlive
from os_ken.lib.packet.bgp import BGPNotification
from os_ken.lib.packet.bgp import BGP_MSG_OPEN
from os_ken.lib.packet.bgp import BGP_MSG_UPDATE
from os_ken.lib.packet.bgp import BGP_MSG_KEEPALIVE
from os_ken.lib.packet.bgp import BGP_MSG_NOTIFICATION
from os_ken.lib.packet.bgp import BGP_MSG_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_FOUR_OCTET_AS_NUMBER
from os_ken.lib.packet.bgp import BGP_CAP_ENHANCED_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_MULTIPROTOCOL
from os_ken.lib.packet.bgp import BGP_ERROR_HOLD_TIMER_EXPIRED
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_HOLD_TIMER_EXPIRED
from os_ken.lib.packet.bgp import get_rf
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import BGP_FSM_CONNECT
from os_ken.services.protocols.bgp.constants import BGP_FSM_OPEN_CONFIRM
from os_ken.services.protocols.bgp.constants import BGP_FSM_OPEN_SENT
from os_ken.services.protocols.bgp.constants import BGP_VERSION_NUM
from os_ken.services.protocols.bgp.protocol import Protocol
def is_local_router_id_greater(self):
    """Compares *True* if local router id is greater when compared to peer
        bgp id.

        Should only be called after protocol has reached OpenConfirm state.
        """
    from os_ken.services.protocols.bgp.utils.bgp import from_inet_ptoi
    if not self.state == BGP_FSM_OPEN_CONFIRM:
        raise BgpProtocolException(desc='Can access remote router id only after open message is received')
    remote_id = self.recv_open_msg.bgp_identifier
    local_id = self.sent_open_msg.bgp_identifier
    return from_inet_ptoi(local_id) > from_inet_ptoi(remote_id)