from __future__ import annotations
import datetime
import random
import struct
from io import BytesIO as _BytesIO
from typing import (
import bson
from bson import CodecOptions, _decode_selective, _dict_to_bson, _make_c_string, encode
from bson.int64 import Int64
from bson.raw_bson import (
from bson.son import SON
from pymongo.errors import (
from pymongo.hello import HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def as_command(self, conn: Connection, apply_timeout: bool=False) -> tuple[SON[str, Any], str]:
    """Return a getMore command document for this query."""
    if self._as_command is not None:
        return self._as_command
    cmd: SON[str, Any] = _gen_get_more_command(self.cursor_id, self.coll, self.ntoreturn, self.max_await_time_ms, self.comment, conn)
    if self.session:
        self.session._apply_to(cmd, False, self.read_preference, conn)
    conn.add_server_api(cmd)
    conn.send_cluster_time(cmd, self.session, self.client)
    client = self.client
    if client._encrypter and (not client._encrypter._bypass_auto_encryption):
        cmd = cast(SON[str, Any], client._encrypter.encrypt(self.db, cmd, self.codec_options))
    if apply_timeout:
        conn.apply_timeout(client, cmd=None)
    self._as_command = (cmd, self.db)
    return self._as_command