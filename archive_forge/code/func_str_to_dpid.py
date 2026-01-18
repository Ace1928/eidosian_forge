def str_to_dpid(dpid_str):
    assert len(dpid_str) == _DPID_LEN
    return int(dpid_str, 16)