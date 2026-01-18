import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
def report_all_anti_patterns(prof, should_benchmark: bool=False, print_enable: bool=True, json_report_dir: Optional[str]=None):
    report_dict: Dict = {}
    anti_patterns = [ExtraCUDACopyPattern(prof, should_benchmark), FP32MatMulPattern(prof, should_benchmark), OptimizerSingleTensorPattern(prof, should_benchmark), SynchronizedDataLoaderPattern(prof, should_benchmark), GradNotSetToNonePattern(prof, should_benchmark), Conv2dBiasFollowedByBatchNorm2dPattern(prof, should_benchmark), MatMulDimInFP16Pattern(prof, should_benchmark)]
    reported = set()
    summaries = []
    message_list = [f'{'-' * 40}TorchTidy Report{'-' * 40}']
    message_list.append('Matched Events:')
    for anti_pattern in anti_patterns:
        matched_events = anti_pattern.matched_events()
        if not matched_events:
            continue
        summaries.append(anti_pattern.summary(matched_events))
        for event in matched_events:
            report_msg = anti_pattern.report(event)
            if report_msg not in reported:
                message_list.append(report_msg)
                reported.add(report_msg)
                src_location, line_no = source_code_location(event).split(':')
                report_dict.setdefault(src_location, []).append({'line_number': int(line_no), 'name': anti_pattern.name, 'url': anti_pattern.url, 'message': anti_pattern.description})
    if json_report_dir is not None:
        json_report_path = os.path.join(json_report_dir, 'torchtidy_report.json')
        if os.path.exists(json_report_path):
            with open(json_report_path) as f:
                exisiting_report = json.load(f)
                exisiting_report.update(report_dict)
                report_dict = exisiting_report
        with open(json_report_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
    message_list.append('Summary:')
    message_list += summaries
    message_list.append(f'{'-' * 40}TorchTidy Report{'-' * 40}')
    if print_enable:
        print('\n'.join(message_list))