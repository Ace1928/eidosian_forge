from ._base import *
import operator as op
@property
def cache_filepath(self):
    return fio.join(self.save_path, self.cache_file)