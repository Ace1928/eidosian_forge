import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def chrome_tracing_dump(self, filename=None):
    """Return a list of profiling events that can viewed as a timeline.

        To view this information as a timeline, simply dump it as a json file
        by passing in "filename" or using using json.dump, and then load go to
        chrome://tracing in the Chrome web browser and load the dumped file.
        Make sure to enable "Flow events" in the "View Options" menu.

        Args:
            filename: If a filename is provided, the timeline is dumped to that
                file.

        Returns:
            If filename is not provided, this returns a list of profiling
                events. Each profile event is a dictionary.
        """
    self._check_connected()
    import time
    time.sleep(1)
    profile_events = self.profile_events()
    all_events = []
    for component_id_hex, component_events in profile_events.items():
        component_type = component_events[0]['component_type']
        if component_type not in ['worker', 'driver']:
            continue
        for event in component_events:
            new_event = {'cat': event['event_type'], 'name': event['event_type'], 'pid': event['node_ip_address'], 'tid': event['component_type'] + ':' + event['component_id'], 'ts': self._nanoseconds_to_microseconds(event['start_time']), 'dur': self._nanoseconds_to_microseconds(event['end_time'] - event['start_time']), 'ph': 'X', 'cname': self._default_color_mapping[event['event_type']], 'args': event['extra_data']}
            if 'cname' in event['extra_data']:
                new_event['cname'] = event['extra_data']['cname']
            if 'name' in event['extra_data']:
                new_event['name'] = event['extra_data']['name']
            all_events.append(new_event)
    if not all_events:
        logger.warning('No profiling events found. Ray profiling must be enabled by setting RAY_PROFILING=1, and make sure RAY_task_events_report_interval_ms is a positive value (default 1000).')
    if filename is not None:
        with open(filename, 'w') as outfile:
            json.dump(all_events, outfile)
    else:
        return all_events