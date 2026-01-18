from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
def set_app_supported_versions(vers):
    global _supported_versions
    _supported_versions &= set(vers)
    assert _supported_versions, 'No OpenFlow version is available'