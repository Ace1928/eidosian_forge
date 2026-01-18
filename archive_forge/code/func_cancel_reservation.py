import abc
@staticmethod
@abc.abstractmethod
def cancel_reservation(context, reservation_id):
    """Cancel a reservation register

        :param context: The request context, for access checks.
        :param reservation_id: ID of the reservation register to cancel.
        """