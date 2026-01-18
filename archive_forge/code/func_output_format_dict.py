import argparse
import json
import sys
import yaml
import openstack.cloud
import openstack.cloud.inventory
from openstack import exceptions
def output_format_dict(data, use_yaml):
    if use_yaml:
        return yaml.safe_dump(data, default_flow_style=False)
    else:
        return json.dumps(data, sort_keys=True, indent=2)