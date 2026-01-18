from openstack.key_manager.v1 import container as _container
from openstack.key_manager.v1 import order as _order
from openstack.key_manager.v1 import secret as _secret
from openstack import proxy
def delete_order(self, order, ignore_missing=True):
    """Delete an order

        :param order: The value can be either the ID of a order or a
            :class:`~openstack.key_manager.v1.order.Order`
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the order does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent order.

        :returns: ``None``
        """
    self._delete(_order.Order, order, ignore_missing=ignore_missing)