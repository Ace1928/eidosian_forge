import codecs
import concurrent.futures
import glob
import os
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import find_command, is_ignored_path, popen_wrapper
def compile_messages(self, locations):
    """
        Locations is a list of tuples: [(directory, file), ...]
        """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, (dirpath, f) in enumerate(locations):
            po_path = Path(dirpath) / f
            mo_path = po_path.with_suffix('.mo')
            try:
                if mo_path.stat().st_mtime >= po_path.stat().st_mtime:
                    if self.verbosity > 0:
                        self.stdout.write('File “%s” is already compiled and up to date.' % po_path)
                    continue
            except FileNotFoundError:
                pass
            if self.verbosity > 0:
                self.stdout.write('processing file %s in %s' % (f, dirpath))
            if has_bom(po_path):
                self.stderr.write('The %s file has a BOM (Byte Order Mark). Django only supports .po files encoded in UTF-8 and without any BOM.' % po_path)
                self.has_errors = True
                continue
            if i == 0 and (not is_writable(mo_path)):
                self.stderr.write('The po files under %s are in a seemingly not writable location. mo files will not be updated/created.' % dirpath)
                self.has_errors = True
                return
            args = [self.program, *self.program_options, '-o', mo_path, po_path]
            futures.append(executor.submit(popen_wrapper, args))
        for future in concurrent.futures.as_completed(futures):
            output, errors, status = future.result()
            if status:
                if self.verbosity > 0:
                    if errors:
                        self.stderr.write('Execution of %s failed: %s' % (self.program, errors))
                    else:
                        self.stderr.write('Execution of %s failed' % self.program)
                self.has_errors = True