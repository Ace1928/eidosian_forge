import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def test_implement_any_one():

    class AnyOneAbc(metaclass=ABCMetaImplementAnyOneOf):

        def _method1_with_2(self):
            return '1-2 ' + self.method2()

        def _method1_with_3(self):
            return '1-3 ' + self.method3()

        def _method2_with_1(self):
            return '2-1 ' + self.method1()

        def _method2_with_3(self):
            return '2-3 ' + self.method3()

        def _method3_with_1(self):
            return '3-1 ' + self.method1()

        def _method3_with_2(self):
            return '3-2 ' + self.method2()

        @alternative(requires='method2', implementation=_method1_with_2)
        @alternative(requires='method3', implementation=_method1_with_3)
        def method1(self):
            """Method1."""

        @alternative(requires='method1', implementation=_method2_with_1)
        @alternative(requires='method3', implementation=_method2_with_3)
        def method2(self):
            """Method2."""

        @alternative(requires='method1', implementation=_method3_with_1)
        @alternative(requires='method2', implementation=_method3_with_2)
        def method3(self):
            """Method3."""

    class Implement1(AnyOneAbc):

        def method1(self):
            """Method1 child."""
            return 'child1'

    class Implement2(AnyOneAbc):

        def method2(self):
            """Method2 child."""
            return 'child2'

    class Implement3(AnyOneAbc):

        def method3(self):
            """Method3 child."""
            return 'child3'
    with pytest.raises(TypeError, match='abstract'):
        AnyOneAbc()
    assert Implement1().method1() == 'child1'
    assert Implement1().method2() == '2-1 child1'
    assert Implement1().method3() == '3-1 child1'
    assert Implement2().method1() == '1-2 child2'
    assert Implement2().method2() == 'child2'
    assert Implement2().method3() == '3-2 child2'
    assert Implement3().method1() == '1-3 child3'
    assert Implement3().method2() == '2-3 child3'
    assert Implement3().method3() == 'child3'