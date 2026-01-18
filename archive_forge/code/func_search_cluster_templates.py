from openstack.cloud import _utils
from openstack import exceptions
def search_cluster_templates(self, name_or_id=None, filters=None, detail=False):
    """Search cluster templates.

        :param name_or_id: cluster template name or ID.
        :param filters: a dict containing additional filters to use.
        :param detail: a boolean to control if we need summarized or
            detailed output.

        :returns: a list of dict containing the cluster templates
        :raises: :class:`~openstack.exceptions.SDKException`: if something goes
            wrong during the OpenStack API call.
        """
    cluster_templates = self.list_cluster_templates(detail=detail)
    return _utils._filter_list(cluster_templates, name_or_id, filters)