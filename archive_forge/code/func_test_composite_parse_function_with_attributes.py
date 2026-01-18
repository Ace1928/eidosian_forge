import unittest
from parameterized import parameterized
import onnx
from onnx import GraphProto, OperatorSetIdProto, checker
@parameterized.expand([('agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu(x) }', {}), ('agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<alpha=2.0>(x) }', {'alpha': 2.0}), ('agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<gamma=3.0>(x) }', {'gamma': 3.0}), ('agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x) }', {'alpha': 2.0, 'gamma': 3.0})])
def test_composite_parse_function_with_attributes(self, graph_text: str, expected_attribute: dict) -> None:
    default_alpha = 1.6732631921768188
    default_gamma = 1.0507010221481323

    def expect_custom_node_attribute(node, attributes):
        for key in attributes:
            match_attr = [attr for attr in node.attribute if attr.name == key]
            assert len(match_attr) == 1
            assert match_attr[0].f == attributes[key]

    def expect_model_function_attribute(model):
        assert len(model.functions[0].attribute_proto) == 2
        attr_proto_alpha = [attr_proto for attr_proto in model.functions[0].attribute_proto if attr_proto.name == 'alpha']
        assert len(attr_proto_alpha) == 1 and attr_proto_alpha[0].f == default_alpha
        attr_proto_gamma = [attr_proto for attr_proto in model.functions[0].attribute_proto if attr_proto.name == 'gamma']
        assert len(attr_proto_gamma) == 1 and attr_proto_gamma[0].f == default_gamma
    function_text = f'\n         <\n         domain: "custom_domain",\n         opset_import: [ "" : 15],\n         doc_string: "Test function proto"\n         >\n           Selu\n           <alpha: float={default_alpha}, gamma: float={default_gamma}>\n           (X) => (C)\n           {{\n               constant_alpha = Constant<value_float: float=@alpha>()\n               constant_gamma = Constant<value_float: float=@gamma>()\n               alpha_x = CastLike(constant_alpha, X)\n               gamma_x = CastLike(constant_gamma, X)\n               exp_x = Exp(X)\n               alpha_x_exp_x = Mul(alpha_x, exp_x)\n               alpha_x_exp_x_ = Sub(alpha_x_exp_x, alpha_x)\n               neg = Mul(gamma_x, alpha_x_exp_x_)\n               pos = Mul(gamma_x, X)\n               _zero = Constant<value_float=0.0>()\n               zero = CastLike(_zero, X)\n               less_eq = LessOrEqual(X, zero)\n               C = Where(less_eq, neg, pos)\n           }}\n        '
    functions = [onnx.parser.parse_function(function_text)]
    graph = onnx.parser.parse_graph(graph_text)
    opset_imports = [OperatorSetIdProto(domain='', version=15), OperatorSetIdProto(domain='custom_domain', version=1)]
    model = onnx.helper.make_model(graph, functions=functions, opset_imports=opset_imports)
    checker.check_model(model)
    expect_model_function_attribute(model)
    expect_custom_node_attribute(model.graph.node[0], expected_attribute)