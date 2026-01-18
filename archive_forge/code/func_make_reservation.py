import abc
@staticmethod
@abc.abstractmethod
def make_reservation(context, project_id, resources, deltas, plugin):
    """Make multiple resource reservations for a given project

        :param context: The request context, for access checks.
        :param resources: A dictionary of the registered resource keys.
        :param project_id: The ID of the project to make the reservations for.
        :return: ``ReservationInfo`` object.
        """