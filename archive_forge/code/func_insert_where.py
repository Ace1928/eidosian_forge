import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
def insert_where(cls, data, where=None):
    """
    Helper for generating conditional INSERT queries.

    For example, prevent INSERTing a new tweet if the user has tweeted within
    the last hour::

        INSERT INTO "tweet" ("user_id", "content", "timestamp")
        SELECT 234, 'some content', now()
        WHERE NOT EXISTS (
            SELECT 1 FROM "tweet"
            WHERE user_id = 234 AND timestamp > now() - interval '1 hour')

    Using this helper:

        cond = ~fn.EXISTS(Tweet.select().where(
            Tweet.user == user_obj,
            Tweet.timestamp > one_hour_ago))

        iq = insert_where(Tweet, {
            Tweet.user: user_obj,
            Tweet.content: 'some content'}, where=cond)

        res = iq.execute()
    """
    for field, default in cls._meta.defaults.items():
        if field.name in data or field in data:
            continue
        value = default() if callable_(default) else default
        data[field] = value
    fields, values = zip(*data.items())
    sq = Select(columns=values).where(where)
    return cls.insert_from(sq, fields).as_rowcount()