from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def introspections(self, **query):
    """Retrieve a generator of introspection records.

        :param dict query: Optional query parameters to be sent to restrict
            the records to be returned. Available parameters include:

            * ``fields``: A list containing one or more fields to be returned
              in the response. This may lead to some performance gain
              because other fields of the resource are not refreshed.
            * ``limit``: Requests at most the specified number of items be
              returned from the query.
            * ``marker``: Specifies the ID of the last-seen introspection. Use
              the ``limit`` parameter to make an initial limited request and
              use the ID of the last-seen introspection from the response as
              the ``marker`` value in a subsequent limited request.
            * ``sort_dir``: Sorts the response by the requested sort direction.
              A valid value is ``asc`` (ascending) or ``desc``
              (descending). Default is ``asc``. You can specify multiple
              pairs of sort key and sort direction query parameters. If
              you omit the sort direction in a pair, the API uses the
              natural sorting direction of the server attribute that is
              provided as the ``sort_key``.
            * ``sort_key``: Sorts the response by the this attribute value.
              Default is ``id``. You can specify multiple pairs of sort
              key and sort direction query parameters. If you omit the
              sort direction in a pair, the API uses the natural sorting
              direction of the server attribute that is provided as the
              ``sort_key``.

        :returns: A generator of :class:`~.introspection.Introspection`
            objects
        """
    return _introspect.Introspection.list(self, **query)