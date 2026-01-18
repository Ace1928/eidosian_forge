import os
import pkg_resources
from paste.script import pluginlib, copydir
from paste.script.command import BadCommand
import subprocess
def load_content(self, base_package, base, name, template, template_renderer=None):
    blank = os.path.join(base, name + '.py')
    read_content = True
    if not os.path.exists(blank):
        if self.use_pkg_resources:
            fullpath = '/'.join([self.source_dir[1], template])
            content = pkg_resources.resource_string(self.source_dir[0], fullpath)
            read_content = False
            blank = fullpath
        else:
            blank = os.path.join(self.source_dir, template)
    if read_content:
        f = open(blank, 'r')
        content = f.read()
        f.close()
    if blank.endswith('_tmpl'):
        content = copydir.substitute_content(content, self.template_vars, filename=blank, template_renderer=template_renderer)
    return content