import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def extend_templates(self, templates, tmpl_name):
    if '#' in tmpl_name:
        dist_name, tmpl_name = tmpl_name.split('#', 1)
    else:
        dist_name, tmpl_name = (None, tmpl_name)
    if dist_name is None:
        for entry in self.all_entry_points():
            if entry.name == tmpl_name:
                tmpl = entry.load()(entry.name)
                dist_name = entry.dist.project_name
                break
        else:
            raise LookupError('Template by name %r not found' % tmpl_name)
    else:
        dist = pkg_resources.get_distribution(dist_name)
        entry = dist.get_entry_info('paste.paster_create_template', tmpl_name)
        tmpl = entry.load()(entry.name)
    full_name = '%s#%s' % (dist_name, tmpl_name)
    for item_full_name, item_tmpl in templates:
        if item_full_name == full_name:
            return
    for req_name in tmpl.required_templates:
        self.extend_templates(templates, req_name)
    templates.append((full_name, tmpl))