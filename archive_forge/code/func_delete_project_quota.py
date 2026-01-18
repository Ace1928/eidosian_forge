import abc
@staticmethod
@abc.abstractmethod
def delete_project_quota(context, project_id):
    """Delete the quota entries for a given project_id.

        After deletion, this project will use default quota values in conf.
        Raise a "not found" error if the quota for the given project was
        never defined.

        :param context: The request context, for access checks.
        :param project_id: The ID of the project to return quotas for.
        """