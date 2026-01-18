import abc
import enum
import threading  # pylint: disable=unused-import
@abc.abstractmethod
def operate(self, group, method, subscription, timeout, initial_metadata=None, payload=None, completion=None, protocol_options=None):
    """Commences an operation.

        Args:
          group: The group identifier of the invoked operation.
          method: The method identifier of the invoked operation.
          subscription: A Subscription to which the results of the operation will be
            passed.
          timeout: A length of time in seconds to allow for the operation.
          initial_metadata: An initial metadata value to be sent to the other side
            of the operation. May be None if the initial metadata will be later
            passed via the returned operator or if there will be no initial metadata
            passed at all.
          payload: An initial payload for the operation.
          completion: A Completion value indicating the end of transmission to the
            other side of the operation.
          protocol_options: A value specified by the provider of a Base interface
            implementation affording custom state and behavior.

        Returns:
          A pair of objects affording information about the operation and action
            continuing the operation. The first element of the returned pair is an
            OperationContext for the operation and the second element of the
            returned pair is an Operator to which operation values not passed in
            this call should later be passed.
        """
    raise NotImplementedError()