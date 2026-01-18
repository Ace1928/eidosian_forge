import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def format_info_string(lm_summary, autoscaler_summary, time=None, gcs_request_time: Optional[float]=None, non_terminated_nodes_time: Optional[float]=None, autoscaler_update_time: Optional[float]=None, verbose: bool=False):
    if time is None:
        time = datetime.now()
    header = '=' * 8 + f' Autoscaler status: {time} ' + '=' * 8
    separator = '-' * len(header)
    if verbose:
        header += '\n'
        if gcs_request_time:
            header += f'GCS request time: {gcs_request_time:3f}s\n'
        if non_terminated_nodes_time:
            header += f'Node Provider non_terminated_nodes time: {non_terminated_nodes_time:3f}s\n'
        if autoscaler_update_time:
            header += f'Autoscaler iteration time: {autoscaler_update_time:3f}s\n'
    available_node_report_lines = []
    if not autoscaler_summary.active_nodes:
        available_node_report = ' (no active nodes)'
    else:
        for node_type, count in autoscaler_summary.active_nodes.items():
            line = f' {count} {node_type}'
            available_node_report_lines.append(line)
        available_node_report = '\n'.join(available_node_report_lines)
    if not autoscaler_summary.idle_nodes:
        idle_node_report = ' (no idle nodes)'
    else:
        idle_node_report_lines = []
        for node_type, count in autoscaler_summary.idle_nodes.items():
            line = f' {count} {node_type}'
            idle_node_report_lines.append(line)
        idle_node_report = '\n'.join(idle_node_report_lines)
    pending_lines = []
    for node_type, count in autoscaler_summary.pending_launches.items():
        line = f' {node_type}, {count} launching'
        pending_lines.append(line)
    for ip, node_type, status in autoscaler_summary.pending_nodes:
        line = f' {ip}: {node_type}, {status.lower()}'
        pending_lines.append(line)
    if pending_lines:
        pending_report = '\n'.join(pending_lines)
    else:
        pending_report = ' (no pending nodes)'
    failure_lines = []
    for ip, node_type in autoscaler_summary.failed_nodes:
        line = f' {node_type}: NodeTerminated (ip: {ip})'
        failure_lines.append(line)
    if autoscaler_summary.node_availability_summary:
        records = sorted(autoscaler_summary.node_availability_summary.node_availabilities.values(), key=lambda record: record.last_checked_timestamp)
        for record in records:
            if record.is_available:
                continue
            assert record.unavailable_node_information is not None
            node_type = record.node_type
            category = record.unavailable_node_information.category
            description = record.unavailable_node_information.description
            attempted_time = datetime.fromtimestamp(record.last_checked_timestamp)
            formatted_time = f'{attempted_time.hour:02d}:{attempted_time.minute:02d}:{attempted_time.second:02d}'
            line = f' {node_type}: {category} (latest_attempt: {formatted_time})'
            if verbose:
                line += f' - {description}'
            failure_lines.append(line)
    failure_lines = failure_lines[:-constants.AUTOSCALER_MAX_FAILURES_DISPLAYED:-1]
    failure_report = 'Recent failures:\n'
    if failure_lines:
        failure_report += '\n'.join(failure_lines)
    else:
        failure_report += ' (no failures)'
    usage_report = get_usage_report(lm_summary, verbose)
    demand_report = get_demand_report(lm_summary)
    formatted_output = f'{header}\nNode status\n{separator}\nActive:\n{available_node_report}'
    if not autoscaler_summary.legacy:
        formatted_output += f'\nIdle:\n{idle_node_report}'
    formatted_output += f'\nPending:\n{pending_report}\n{failure_report}\n\nResources\n{separator}\n{('Total ' if verbose else '')}Usage:\n{usage_report}\n{('Total ' if verbose else '')}Demands:\n{demand_report}'
    if verbose:
        if lm_summary.usage_by_node:
            formatted_output += get_per_node_breakdown(lm_summary, autoscaler_summary.node_type_mapping, autoscaler_summary.node_activities, verbose)
        else:
            formatted_output += '\n'
    return formatted_output.strip()