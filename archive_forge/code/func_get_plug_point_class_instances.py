from oslo_log import log as logging
from heat.engine import resources
def get_plug_point_class_instances():
    """Instances of classes that implements pre/post stack operation methods.

    Get list of instances of classes that (may) implement pre and post
    stack operation methods.

    The list of class instances is sorted using get_ordinal methods
    on the plug point classes. If class1.ordinal() < class2.ordinal(),
    then class1 will be before class2 in the list.
    """
    global pp_class_instances
    if pp_class_instances is None:
        pp_class_instances = []
        pp_classes = []
        try:
            slps = resources.global_env().get_stack_lifecycle_plugins()
            pp_classes = [cls for name, cls in slps]
        except Exception:
            LOG.exception('failed to get lifecycle plug point classes')
        for ppc in pp_classes:
            try:
                pp_class_instances.append(ppc())
            except Exception:
                LOG.exception('failed to instantiate stack lifecycle class %s', ppc)
        try:
            pp_class_instances = sorted(pp_class_instances, key=lambda ppci: ppci.get_ordinal())
        except Exception:
            LOG.exception('failed to sort lifecycle plug point classes')
    return pp_class_instances