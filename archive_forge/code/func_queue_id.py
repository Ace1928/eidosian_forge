from __future__ import annotations
import datetime
from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Index, Integer,
from sqlalchemy.orm import relationship
from sqlalchemy.schema import MetaData
@declared_attr
def queue_id(self):
    return Column(Integer, ForeignKey('%s.id' % class_registry['Queue'].__tablename__, name='FK_kombu_message_queue'))