import base64
import uuid
from heat.common.i18n import _
def byte_at(off):
    return (value >> off if off >= 0 else value << -off) & 255