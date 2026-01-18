import subprocess, sys, os
from distutils.core import Command
from distutils.debug import DEBUG
from distutils.file_util import write_file
from distutils.errors import *
from distutils.sysconfig import get_python_version
from distutils import log
def finalize_package_data(self):
    self.ensure_string('group', 'Development/Libraries')
    self.ensure_string('vendor', '%s <%s>' % (self.distribution.get_contact(), self.distribution.get_contact_email()))
    self.ensure_string('packager')
    self.ensure_string_list('doc_files')
    if isinstance(self.doc_files, list):
        for readme in ('README', 'README.txt'):
            if os.path.exists(readme) and readme not in self.doc_files:
                self.doc_files.append(readme)
    self.ensure_string('release', '1')
    self.ensure_string('serial')
    self.ensure_string('distribution_name')
    self.ensure_string('changelog')
    self.changelog = self._format_changelog(self.changelog)
    self.ensure_filename('icon')
    self.ensure_filename('prep_script')
    self.ensure_filename('build_script')
    self.ensure_filename('install_script')
    self.ensure_filename('clean_script')
    self.ensure_filename('verify_script')
    self.ensure_filename('pre_install')
    self.ensure_filename('post_install')
    self.ensure_filename('pre_uninstall')
    self.ensure_filename('post_uninstall')
    self.ensure_string_list('provides')
    self.ensure_string_list('requires')
    self.ensure_string_list('conflicts')
    self.ensure_string_list('build_requires')
    self.ensure_string_list('obsoletes')
    self.ensure_string('force_arch')