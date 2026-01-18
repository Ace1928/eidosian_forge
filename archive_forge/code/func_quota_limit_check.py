import abc
@staticmethod
@abc.abstractmethod
def quota_limit_check(context, project_id, resources, deltas):
    """Check the current resource usage against a set of deltas.

        This method will check if the provided resource deltas could be
        assigned depending on the current resource usage and the quota limits.
        If the resource deltas plus the resource usage fit under the quota
        limit, the method will pass. If not, a ``OverQuota`` will be raised.

        :param context: The request context, for access checks.
        :param project_id: The ID of the project to make the reservations for.
        :param resources: A dictionary of the registered resource.
        :param deltas: A dictionary of the values to check against the
                       quota limits.
        :return: None if passed; ``OverQuota`` if quota limits are exceeded,
                 ``InvalidQuotaValue`` if delta values are invalid.
        """