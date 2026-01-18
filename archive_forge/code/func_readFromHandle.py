from zope.interface import Interface
def readFromHandle(bufflist, evt):
    """
        Read into the given buffers from this handle.

        @param bufflist: the buffers to read into
        @type bufflist: list of objects implementing the read/write buffer protocol

        @param evt: an IOCP Event object

        @return: tuple (return code, number of bytes read)
        """