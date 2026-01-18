from __future__ import annotations
import copy
from collections.abc import MutableMapping
from itertools import islice
from typing import (
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from pymongo import _csot, common
from pymongo.client_session import ClientSession, _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES, _get_wce_doc
from pymongo.message import (
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def execute_op_msg_no_results(self, conn: Connection, generator: Iterator[Any]) -> None:
    """Execute write commands with OP_MSG and w=0 writeConcern, unordered."""
    db_name = self.collection.database.name
    client = self.collection.database.client
    listeners = client._event_listeners
    op_id = _randint()
    if not self.current_run:
        self.current_run = next(generator)
    run = self.current_run
    while run:
        cmd_name = _COMMANDS[run.op_type]
        bwc = self.bulk_ctx_class(db_name, cmd_name, conn, op_id, listeners, None, run.op_type, self.collection.codec_options)
        while run.idx_offset < len(run.ops):
            cmd = SON([(cmd_name, self.collection.name), ('ordered', False), ('writeConcern', {'w': 0})])
            conn.add_server_api(cmd)
            ops = islice(run.ops, run.idx_offset, None)
            to_send = bwc.execute_unack(cmd, ops, client)
            run.idx_offset += len(to_send)
        self.current_run = run = next(generator, None)