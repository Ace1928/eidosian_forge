from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def mask_config_blocks_from_diff(config, candidate, force_diff_prefix):
    conf_lines = config.split('\n')
    candidate_lines = candidate.split('\n')
    for regex in CONFIG_BLOCKS_FORCED_IN_DIFF:
        block_index_start_end = []
        start_index = None
        for index, line in enumerate(candidate_lines):
            startre = regex['start'].search(line)
            if startre and startre.group(0):
                start_index = index
            else:
                endre = regex['end'].search(line)
                if endre and endre.group(0) and start_index:
                    end_index = index
                    new_block = True
                    for prev_start, prev_end in block_index_start_end:
                        if start_index == prev_start:
                            new_block = False
                            break
                    if new_block and end_index:
                        block_index_start_end.append((start_index, end_index))
        for start, end in block_index_start_end:
            diff = False
            if candidate_lines[start] in conf_lines:
                run_conf_start_index = conf_lines.index(candidate_lines[start])
            else:
                diff = False
                continue
            for i in range(start, end + 1):
                if conf_lines[run_conf_start_index] == candidate_lines[i]:
                    run_conf_start_index = run_conf_start_index + 1
                else:
                    diff = True
                    break
            if diff:
                run_conf_start_index = conf_lines.index(candidate_lines[start])
                for i in range(start, end + 1):
                    conf_lines[run_conf_start_index] = conf_lines[run_conf_start_index] + force_diff_prefix
                    run_conf_start_index = run_conf_start_index + 1
    conf = '\n'.join(conf_lines)
    return conf