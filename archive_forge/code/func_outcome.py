import abc
import enum
import threading  # pylint: disable=unused-import
@abc.abstractmethod
def outcome(self):
    """Indicates the operation's outcome (or that the operation is ongoing).

        Returns:
          None if the operation is still active or the Outcome value for the
            operation if it has terminated.
        """
    raise NotImplementedError()