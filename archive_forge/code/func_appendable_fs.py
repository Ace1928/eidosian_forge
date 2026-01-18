from ._base import *
@property
def appendable_fs(self):
    return [FileExtIO.TXT, FileExtIO.JSONLINES, FileExtIO.TFRECORDS]