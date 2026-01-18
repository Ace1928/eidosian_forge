import abc
import enum
import threading  # pylint: disable=unused-import
@abc.abstractmethod
def operation_stats(self):
    """Reports the number of terminated operations broken down by outcome.

        Returns:
          A dictionary from Outcome.Kind value to an integer identifying the number
            of operations that terminated with that outcome kind.
        """
    raise NotImplementedError()