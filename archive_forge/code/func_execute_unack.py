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
def execute_unack(self, cmd: MutableMapping[str, Any], docs: list[Mapping[str, Any]], client: MongoClient) -> list[Mapping[str, Any]]:
    batched_cmd, to_send = self.__batch_command(cmd, docs)
    self.conn.command(self.db_name, batched_cmd, write_concern=WriteConcern(w=0), session=self.session, client=client)
    return to_send