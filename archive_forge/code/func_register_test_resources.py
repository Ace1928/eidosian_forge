import os
import sys
import fixtures
from oslo_config import cfg
from oslo_log import log as logging
import testscenarios
import testtools
from heat.common import context
from heat.common import messaging
from heat.common import policy
from heat.engine.clients.os import barbican
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.engine.clients.os.neutron import neutron_constraints as neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.clients.os import trove
from heat.engine import environment
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def register_test_resources(self):
    resource._register_class('GenericResourceType', generic_rsrc.GenericResource)
    resource._register_class('MultiStepResourceType', generic_rsrc.MultiStepResource)
    resource._register_class('ResWithShowAttrType', generic_rsrc.ResWithShowAttr)
    resource._register_class('SignalResourceType', generic_rsrc.SignalResource)
    resource._register_class('ResourceWithPropsType', generic_rsrc.ResourceWithProps)
    resource._register_class('ResourceWithPropsRefPropOnDelete', generic_rsrc.ResourceWithPropsRefPropOnDelete)
    resource._register_class('ResourceWithPropsRefPropOnValidate', generic_rsrc.ResourceWithPropsRefPropOnValidate)
    resource._register_class('StackUserResourceType', generic_rsrc.StackUserResource)
    resource._register_class('ResourceWithResourceIDType', generic_rsrc.ResourceWithResourceID)
    resource._register_class('ResourceWithAttributeType', generic_rsrc.ResourceWithAttributeType)
    resource._register_class('ResourceWithRequiredProps', generic_rsrc.ResourceWithRequiredProps)
    resource._register_class('ResourceWithMultipleRequiredProps', generic_rsrc.ResourceWithMultipleRequiredProps)
    resource._register_class('ResourceWithRequiredPropsAndEmptyAttrs', generic_rsrc.ResourceWithRequiredPropsAndEmptyAttrs)
    resource._register_class('ResourceWithPropsAndAttrs', generic_rsrc.ResourceWithPropsAndAttrs)
    (resource._register_class('ResWithStringPropAndAttr', generic_rsrc.ResWithStringPropAndAttr),)
    resource._register_class('ResWithComplexPropsAndAttrs', generic_rsrc.ResWithComplexPropsAndAttrs)
    resource._register_class('ResourceWithCustomConstraint', generic_rsrc.ResourceWithCustomConstraint)
    resource._register_class('ResourceWithComplexAttributesType', generic_rsrc.ResourceWithComplexAttributes)
    resource._register_class('ResourceWithDefaultClientName', generic_rsrc.ResourceWithDefaultClientName)
    resource._register_class('OverwrittenFnGetAttType', generic_rsrc.ResourceWithFnGetAttType)
    resource._register_class('OverwrittenFnGetRefIdType', generic_rsrc.ResourceWithFnGetRefIdType)
    resource._register_class('ResourceWithListProp', generic_rsrc.ResourceWithListProp)
    resource._register_class('StackResourceType', generic_rsrc.StackResourceType)
    resource._register_class('ResourceWithRestoreType', generic_rsrc.ResourceWithRestoreType)
    resource._register_class('ResourceTypeUnSupportedLiberty', generic_rsrc.ResourceTypeUnSupportedLiberty)
    resource._register_class('ResourceTypeSupportedKilo', generic_rsrc.ResourceTypeSupportedKilo)
    resource._register_class('ResourceTypeHidden', generic_rsrc.ResourceTypeHidden)
    resource._register_class('ResourceWithHiddenPropertyAndAttribute', generic_rsrc.ResourceWithHiddenPropertyAndAttribute)