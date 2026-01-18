import time
import os
def submit_path(self, path, tags=None, ignore_dirs=None, cb=None, num_cb=0, status=False, prefix='/'):
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    path = os.path.abspath(path)
    total = 0
    metadata = {}
    if tags:
        metadata['Tags'] = tags
    l = []
    for t in time.gmtime():
        l.append(str(t))
    metadata['Batch'] = '_'.join(l)
    if self.output_domain:
        self.output_domain.put_attributes(metadata['Batch'], {'type': 'Batch'})
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if ignore_dirs:
                for ignore in ignore_dirs:
                    if ignore in dirs:
                        dirs.remove(ignore)
            for file in files:
                fullpath = os.path.join(root, file)
                if status:
                    print('Submitting %s' % fullpath)
                self.submit_file(fullpath, metadata, cb, num_cb, prefix)
                total += 1
    elif os.path.isfile(path):
        self.submit_file(path, metadata, cb, num_cb)
        total += 1
    else:
        print('problem with %s' % path)
    return (metadata['Batch'], total)