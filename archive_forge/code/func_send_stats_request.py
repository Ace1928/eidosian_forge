import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def send_stats_request(dp, stats, waiters, msgs, logger=None):
    dp.set_xid(stats)
    waiters_per_dp = waiters.setdefault(dp.id, {})
    lock = hub.Event()
    previous_msg_len = len(msgs)
    waiters_per_dp[stats.xid] = (lock, msgs)
    send_msg(dp, stats, logger)
    lock.wait(timeout=DEFAULT_TIMEOUT)
    current_msg_len = len(msgs)
    while current_msg_len > previous_msg_len:
        previous_msg_len = current_msg_len
        lock.wait(timeout=DEFAULT_TIMEOUT)
        current_msg_len = len(msgs)
    if not lock.is_set():
        del waiters_per_dp[stats.xid]