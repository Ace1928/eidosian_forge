def get_ordinal(self):
    """Get the sort order for pre and post operation execution.

        The values returned by get_ordinal are used to create a partial order
        for pre and post operation method invocations. The default ordinal
        value of 100 may be overridden.
        If class1inst.ordinal() < class2inst.ordinal(), then the method on
        class1inst will be executed before the method on class2inst.
        If class1inst.ordinal() > class2inst.ordinal(), then the method on
        class1inst will be executed after the method on class2inst.
        If class1inst.ordinal() == class2inst.ordinal(), then the order of
        method invocation is indeterminate.
        """
    return 100