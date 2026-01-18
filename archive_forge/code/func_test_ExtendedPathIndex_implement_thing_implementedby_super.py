import unittest
def test_ExtendedPathIndex_implement_thing_implementedby_super(self):
    from zope.interface import ro

    class _Based:
        __bases__ = ()

        def __init__(self, name, bases=(), attrs=None):
            self.__name__ = name
            self.__bases__ = bases

        def __repr__(self):
            return self.__name__
    Interface = _Based('Interface', (), {})

    class IPluggableIndex(Interface):
        pass

    class ILimitedResultIndex(IPluggableIndex):
        pass

    class IQueryIndex(IPluggableIndex):
        pass

    class IPathIndex(Interface):
        pass
    obj = _Based('object')
    PathIndex = _Based('PathIndex', (IPathIndex, IQueryIndex, obj))
    ExtendedPathIndex = _Based('ExtendedPathIndex', (ILimitedResultIndex, IQueryIndex, PathIndex))
    result = self._callFUT(ExtendedPathIndex, log_changed_ro=True, strict=False)
    self.assertEqual(result, [ExtendedPathIndex, ILimitedResultIndex, PathIndex, IPathIndex, IQueryIndex, IPluggableIndex, Interface, obj])
    record, = self.log_handler.records
    self.assertIn('used the legacy', record.getMessage())
    with self.assertRaises(ro.InconsistentResolutionOrderError):
        self._callFUT(ExtendedPathIndex, strict=True)