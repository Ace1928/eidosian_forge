from apitools.base.py import exceptions
@property
def stream_exhausted(self):
    return self.__stream_at_end