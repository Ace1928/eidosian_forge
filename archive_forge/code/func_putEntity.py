from twisted.python import reflect
def putEntity(self, name, entity):
    """Store an entity if it meets both constraints.

        Otherwise raise a ConstraintViolation.
        """
    if self.nameConstraint(name):
        if self.entityConstraint(entity):
            self.reallyPutEntity(name, entity)
        else:
            raise ConstraintViolation('Entity constraint violated.')
    else:
        raise ConstraintViolation('Name constraint violated.')