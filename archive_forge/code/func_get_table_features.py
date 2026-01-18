import logging
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_4_parser
from os_ken.lib import ofctl_utils
def get_table_features(dp, waiters, to_user=True):
    stats = dp.ofproto_parser.OFPTableFeaturesStatsRequest(dp, 0, [])
    msgs = []
    ofproto = dp.ofproto
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    p_type_instructions = [ofproto.OFPTFPT_INSTRUCTIONS, ofproto.OFPTFPT_INSTRUCTIONS_MISS]
    p_type_next_tables = [ofproto.OFPTFPT_NEXT_TABLES, ofproto.OFPTFPT_NEXT_TABLES_MISS, ofproto.OFPTFPT_TABLE_SYNC_FROM]
    p_type_actions = [ofproto.OFPTFPT_WRITE_ACTIONS, ofproto.OFPTFPT_WRITE_ACTIONS_MISS, ofproto.OFPTFPT_APPLY_ACTIONS, ofproto.OFPTFPT_APPLY_ACTIONS_MISS]
    p_type_oxms = [ofproto.OFPTFPT_MATCH, ofproto.OFPTFPT_WILDCARDS, ofproto.OFPTFPT_WRITE_SETFIELD, ofproto.OFPTFPT_WRITE_SETFIELD_MISS, ofproto.OFPTFPT_APPLY_SETFIELD, ofproto.OFPTFPT_APPLY_SETFIELD_MISS]
    p_type_experimenter = [ofproto.OFPTFPT_EXPERIMENTER, ofproto.OFPTFPT_EXPERIMENTER_MISS]
    tables = []
    for msg in msgs:
        stats = msg.body
        for stat in stats:
            s = stat.to_jsondict()[stat.__class__.__name__]
            properties = []
            for prop in stat.properties:
                p = {}
                t = UTIL.ofp_table_feature_prop_type_to_user(prop.type)
                p['type'] = t if t != prop.type else 'UNKNOWN'
                if prop.type in p_type_instructions:
                    instruction_ids = []
                    for i in prop.instruction_ids:
                        inst = {'len': i.len, 'type': i.type}
                        instruction_ids.append(inst)
                    p['instruction_ids'] = instruction_ids
                elif prop.type in p_type_next_tables:
                    table_ids = []
                    for i in prop.table_ids:
                        table_ids.append(i)
                    p['table_ids'] = table_ids
                elif prop.type in p_type_actions:
                    action_ids = []
                    for i in prop.action_ids:
                        act = i.to_jsondict()[i.__class__.__name__]
                        action_ids.append(act)
                    p['action_ids'] = action_ids
                elif prop.type in p_type_oxms:
                    oxm_ids = []
                    for i in prop.oxm_ids:
                        oxm = i.to_jsondict()[i.__class__.__name__]
                        oxm_ids.append(oxm)
                    p['oxm_ids'] = oxm_ids
                elif prop.type in p_type_experimenter:
                    pass
                properties.append(p)
            s['name'] = stat.name.decode('utf-8')
            s['properties'] = properties
            if to_user:
                s['table_id'] = UTIL.ofp_table_to_user(stat.table_id)
            tables.append(s)
    return wrap_dpid_dict(dp, tables, to_user)