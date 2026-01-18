import logging
def batch_writer(self, overwrite_by_pkeys=None):
    """Create a batch writer object.

        This method creates a context manager for writing
        objects to Amazon DynamoDB in batch.

        The batch writer will automatically handle buffering and sending items
        in batches.  In addition, the batch writer will also automatically
        handle any unprocessed items and resend them as needed.  All you need
        to do is call ``put_item`` for any items you want to add, and
        ``delete_item`` for any items you want to delete.

        Example usage::

            with table.batch_writer() as batch:
                for _ in xrange(1000000):
                    batch.put_item(Item={'HashKey': '...',
                                         'Otherstuff': '...'})
                # You can also delete_items in a batch.
                batch.delete_item(Key={'HashKey': 'SomeHashKey'})

        :type overwrite_by_pkeys: list(string)
        :param overwrite_by_pkeys: De-duplicate request items in buffer
            if match new request item on specified primary keys. i.e
            ``["partition_key1", "sort_key2", "sort_key3"]``

        """
    return BatchWriter(self.name, self.meta.client, overwrite_by_pkeys=overwrite_by_pkeys)