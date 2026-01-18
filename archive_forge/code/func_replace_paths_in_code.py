import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def replace_paths_in_code(self, co):
    new_filename = original_filename = os.path.normpath(co.co_filename)
    for f, r in self.replace_paths:
        if original_filename.startswith(f):
            new_filename = r + original_filename[len(f):]
            break
    if self.debug and original_filename not in self.processed_paths:
        if new_filename != original_filename:
            self.msgout(2, 'co_filename %r changed to %r' % (original_filename, new_filename))
        else:
            self.msgout(2, 'co_filename %r remains unchanged' % (original_filename,))
        self.processed_paths.append(original_filename)
    consts = list(co.co_consts)
    for i in range(len(consts)):
        if isinstance(consts[i], type(co)):
            consts[i] = self.replace_paths_in_code(consts[i])
    return co.replace(co_consts=tuple(consts), co_filename=new_filename)