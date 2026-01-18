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
def chrome_tracing_object_transfer_dump(self, filename=None):
    """Return a list of transfer events that can viewed as a timeline.

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
    node_id_to_address = {}
    for node_info in self.node_table():
        node_id_to_address[node_info['NodeID']] = '{}:{}'.format(node_info['NodeManagerAddress'], node_info['ObjectManagerPort'])
    all_events = []
    for key, items in self.profile_events().items():
        if items[0]['component_type'] != 'object_manager':
            continue
        for event in items:
            if event['event_type'] == 'transfer_send':
                object_ref, remote_node_id, _, _ = event['extra_data']
            elif event['event_type'] == 'transfer_receive':
                object_ref, remote_node_id, _ = event['extra_data']
            elif event['event_type'] == 'receive_pull_request':
                object_ref, remote_node_id = event['extra_data']
            else:
                assert False, 'This should be unreachable.'
            object_ref_int = int(object_ref[:2], 16)
            color = self._chrome_tracing_colors[object_ref_int % len(self._chrome_tracing_colors)]
            new_event = {'cat': event['event_type'], 'name': event['event_type'], 'pid': node_id_to_address[key], 'tid': node_id_to_address[remote_node_id], 'ts': self._nanoseconds_to_microseconds(event['start_time']), 'dur': self._nanoseconds_to_microseconds(event['end_time'] - event['start_time']), 'ph': 'X', 'cname': color, 'args': event['extra_data']}
            all_events.append(new_event)
            if event['event_type'] == 'transfer_send':
                additional_event = new_event.copy()
                additional_event['cname'] = 'black'
                all_events.append(additional_event)
            elif event['event_type'] == 'transfer_receive':
                additional_event = new_event.copy()
                additional_event['cname'] = 'grey'
                all_events.append(additional_event)
            else:
                pass
    if filename is not None:
        with open(filename, 'w') as outfile:
            json.dump(all_events, outfile)
    else:
        return all_events