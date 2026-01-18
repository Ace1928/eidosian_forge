from kazoo import client
from kazoo import exceptions as k_exc
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow import logging
def checked_commit(txn):
    """Commits a kazoo transcation and validates the result.

    NOTE(harlowja): Until https://github.com/python-zk/kazoo/pull/224 is fixed
    or a similar pull request is merged we have to workaround the transaction
    failing silently.
    """
    if not txn.operations:
        return []
    results = txn.commit()
    failures = []
    for op, result in zip(txn.operations, results):
        if isinstance(result, k_exc.KazooException):
            failures.append((op, result))
    if len(results) < len(txn.operations):
        raise KazooTransactionException('Transaction returned %s results, this is less than the number of expected transaction operations %s' % (len(results), len(txn.operations)), failures)
    if len(results) > len(txn.operations):
        raise KazooTransactionException('Transaction returned %s results, this is greater than the number of expected transaction operations %s' % (len(results), len(txn.operations)), failures)
    if failures:
        raise KazooTransactionException('Transaction with %s operations failed: %s' % (len(txn.operations), prettify_failures(failures, limit=1)), failures)
    return results