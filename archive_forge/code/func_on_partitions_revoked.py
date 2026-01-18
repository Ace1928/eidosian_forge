import abc
@abc.abstractmethod
def on_partitions_revoked(self, revoked):
    """
        A coroutine or function the user can implement to provide cleanup or
        custom state save on the start of a rebalance operation.

        This method will be called *before* a rebalance operation starts and
        *after* the consumer stops fetching data.

        If you are using manual commit you have to commit all consumed offsets
        here, to avoid duplicate message delivery after rebalance is finished.

        .. note:: This method is only called before rebalances. It is not
            called prior to :meth:`.AIOKafkaConsumer.stop`

        Arguments:
            revoked (list(TopicPartition)): the partitions that were assigned
                to the consumer on the last rebalance
        """
    pass