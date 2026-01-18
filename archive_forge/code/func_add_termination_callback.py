import abc
import enum
import threading  # pylint: disable=unused-import
@abc.abstractmethod
def add_termination_callback(self, callback):
    """Adds a function to be called upon operation termination.

        Args:
          callback: A callable to be passed an Outcome value on operation
            termination.

        Returns:
          None if the operation has not yet terminated and the passed callback will
            later be called when it does terminate, or if the operation has already
            terminated an Outcome value describing the operation termination and the
            passed callback will not be called as a result of this method call.
        """
    raise NotImplementedError()