from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def getproperties(self, filename, convert_time=False, no_conversion=None):
    """
        Return properties described in substream.

        :param filename: path of stream in storage tree (see openstream for syntax)
        :param convert_time: bool, if True timestamps will be converted to Python datetime
        :param no_conversion: None or list of int, timestamps not to be converted
            (for example total editing time is not a real timestamp)

        :returns: a dictionary of values indexed by id (integer)
        """
    if no_conversion == None:
        no_conversion = []
    streampath = filename
    if not isinstance(streampath, str):
        streampath = '/'.join(streampath)
    fp = self.openstream(filename)
    data = {}
    try:
        s = fp.read(28)
        clsid = _clsid(s[8:24])
        s = fp.read(20)
        fmtid = _clsid(s[:16])
        fp.seek(i32(s, 16))
        s = b'****' + fp.read(i32(fp.read(4)) - 4)
        num_props = i32(s, 4)
    except BaseException as exc:
        msg = 'Error while parsing properties header in stream {}: {}'.format(repr(streampath), exc)
        self._raise_defect(DEFECT_INCORRECT, msg, type(exc))
        return data
    num_props = min(num_props, int(len(s) / 8))
    for i in iterrange(num_props):
        property_id = 0
        try:
            property_id = i32(s, 8 + i * 8)
            offset = i32(s, 12 + i * 8)
            property_type = i32(s, offset)
            vt_name = VT.get(property_type, 'UNKNOWN')
            log.debug('property id=%d: type=%d/%s offset=%X' % (property_id, property_type, vt_name, offset))
            value = self._parse_property(s, offset + 4, property_id, property_type, convert_time, no_conversion)
            data[property_id] = value
        except BaseException as exc:
            msg = 'Error while parsing property id %d in stream %s: %s' % (property_id, repr(streampath), exc)
            self._raise_defect(DEFECT_INCORRECT, msg, type(exc))
    return data