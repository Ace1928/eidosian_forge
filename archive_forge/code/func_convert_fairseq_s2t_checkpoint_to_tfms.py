import argparse
import torch
from torch import nn
from transformers import Speech2TextConfig, Speech2TextForConditionalGeneration
def convert_fairseq_s2t_checkpoint_to_tfms(checkpoint_path, pytorch_dump_folder_path):
    m2m_100 = torch.load(checkpoint_path, map_location='cpu')
    args = m2m_100['args']
    state_dict = m2m_100['model']
    lm_head_weights = state_dict['decoder.output_projection.weight']
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    vocab_size = state_dict['decoder.embed_tokens.weight'].shape[0]
    tie_embeds = args.share_decoder_input_output_embed
    conv_kernel_sizes = [int(i) for i in args.conv_kernel_sizes.split(',')]
    config = Speech2TextConfig(vocab_size=vocab_size, max_source_positions=args.max_source_positions, max_target_positions=args.max_target_positions, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_embed_dim, decoder_ffn_dim=args.decoder_ffn_embed_dim, d_model=args.encoder_embed_dim, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_function='relu', num_conv_layers=len(conv_kernel_sizes), conv_channels=args.conv_channels, conv_kernel_sizes=conv_kernel_sizes, input_feat_per_channel=args.input_feat_per_channel, input_channels=args.input_channels, tie_word_embeddings=tie_embeds, num_beams=5, max_length=200, use_cache=True, decoder_start_token_id=2, early_stopping=True)
    model = Speech2TextForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and (not set(missing) <= {'encoder.embed_positions.weights', 'decoder.embed_positions.weights'}):
        raise ValueError(f'Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing, but all the following weights are missing {missing}')
    if tie_embeds:
        model.lm_head = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.lm_head.weight.data = lm_head_weights
    model.save_pretrained(pytorch_dump_folder_path)