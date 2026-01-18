import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def print_entry_points_by_group(self, group, ep_pat):
    env = pkg_resources.Environment()
    project_names = sorted(env)
    for project_name in project_names:
        dists = list(env[project_name])
        assert dists
        dist = dists[0]
        entries = list(dist.get_entry_map(group).values())
        if ep_pat:
            entries = [e for e in entries if ep_pat.search(e.name)]
        if not entries:
            continue
        if len(dists) > 1:
            print('%s (+ %i older versions)' % (dist, len(dists) - 1))
        else:
            print('%s' % dist)
        entries.sort(key=lambda entry: entry.name)
        for entry in entries:
            print(self._ep_description(entry))
            desc = self.get_entry_point_description(entry, group)
            if desc and desc.description:
                print(self.wrap(desc.description, indent=4))