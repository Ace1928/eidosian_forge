import abc
@abc.abstractmethod
def consume_and_terminate(self, value):
    """Supplies a value and signals that no more values will be supplied.

        Args:
          value: Any value accepted by this Consumer.
        """
    raise NotImplementedError()